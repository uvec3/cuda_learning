#pragma once

#include <assert.h>
#include <boost/pfr.hpp>
#include <type_traits>
#include <tuple>
#include <cstddef>
#include <iterator>

template <std::size_t N, typename Func>
constexpr void compile_time_for(Func&& f)
{
    [=]<std::size_t... Is>(std::index_sequence<Is...>) {
        (f(std::integral_constant<std::size_t, Is>{}), ...); // C++17 fold expression
    }(std::make_index_sequence<N>{});
}

template <typename T>
struct is_tuple_like : std::false_type {};

template <typename... Ts>
struct is_tuple_like<std::tuple<Ts...>> : std::true_type {};

template <typename T>
concept IsTuple = is_tuple_like<std::remove_cvref_t<T>>::value;

template <typename T>
concept TupleLike =
    requires
    {
        typename std::tuple_size<std::remove_cvref_t<T>>::type;
    } &&
    requires(T t)
    {
        std::get<0>(t); // Checks for std::get
    };


template <typename T>
concept ArrayLike =
    requires(T t)
    {
        { t[0] }; // Can be indexed
        { std::size(t) } -> std::convertible_to<std::size_t>; // Has std::size
    };


template <typename T>
constexpr int generic_tuple_size()
{
    if constexpr (TupleLike<T>)//for tuples, pairs
        return std::tuple_size_v<T>;
    else if constexpr (std::is_aggregate_v<T>)//for aggregates (e.g: struct or class)
        return boost::pfr::tuple_size_v<T>;
    else
        return 0;
}


template <std::size_t I, typename T>
constexpr decltype(auto) generic_get(T&& v) {
    if constexpr (TupleLike<T>)
        return std::get<I>(std::forward<T>(v));
    else
        return boost::pfr::get<I>(std::forward<T>(v));
}


template <typename Callback, typename View>
void for_each_ptr_in_soa_recursive(View& view, Callback callback)
{
    if constexpr (std::is_pointer_v<View>)//a single pointer view
    {
        callback(view);
    }
    else if constexpr (std::ranges::range<View>)//aray like view type
    {
        for (auto& field:view)
        {
            for_each_ptr_in_soa_recursive(field,callback);
        }
    }
    else if constexpr (std::is_class<std::remove_reference_t<decltype(view)>>::value)//struct like view type
    {
        constexpr int N=generic_tuple_size<View>();
        printf("Size: %d\n", N);
        compile_time_for<N>([&](auto i_c)//compile time for is needed to get constexpr i inside
        {
            constexpr auto i=i_c();
            auto& field=generic_get<i>(view);
            for_each_ptr_in_soa_recursive(field,callback);
        });
    }
}

template <typename View>
void soa_malloc(View& view, size_t size, const std::function<void*(size_t)>& allocate_callback= malloc)
{
    auto allocate_cb = [&]<typename T>(T*& ptr)
    {
        ptr=static_cast<T*>(allocate_callback(size*sizeof(T)));
        printf("%llu bytes allocated\n",size*sizeof(T));
    };
    for_each_ptr_in_soa_recursive(view,allocate_cb);
}

template <typename View>
void soa_free(View& view, const std::function<void(void*)>& free_callback=free)
{
    auto free_cb = [&]<typename T>(T*& ptr)
    {
        free_callback(ptr);
        ptr=nullptr;
    };
    for_each_ptr_in_soa_recursive(view,free_cb);
}

template <typename View>
void soa_realloc(View& view, size_t size, const std::function<void(void*,size_t)>& realloc_callback= realloc)
{
    auto realloc_cb = [&]<typename T>(T*& ptr)
    {
        realloc_callback(ptr,size*sizeof(T));
        ptr=nullptr;
    };
    for_each_ptr_in_soa_recursive(view,realloc_cb);
}

template <typename View>
void soa_memcpy(View& dst, View& src, size_t size)
{
    auto copy_cb = [&]<typename T>(T*& dst_ptr)
    {
        int64_t diff=static_cast<char*>(dst_ptr) - static_cast<char* >(&dst);
        assert(diff>=0 && diff<sizeof(View));//pointer is within the view
        char* src_base=static_cast<char*>(&src);
        void* src_ptr=src_base+diff;
        memcpy(dst_ptr,src_ptr,size*sizeof(T));
    };
    for_each_ptr_in_soa_recursive(dst,copy_cb);
}


template<typename View>
class soa_vector
{
private:
    View m_view{};
    size_t m_capacity;
    size_t m_size;


public:

    soa_vector(size_t size):m_size{size}
    {
        m_capacity=m_size;
        soa_malloc(m_view,m_capacity);
    }

    ~soa_vector()
    {
        free();
    }

    const View& view()
    {
        return m_view;
    }

    size_t size()
    {
        return m_size;
    }

    void resize(size_t size)
    {
        View& view=*this;
        if (m_size!=size)
        {
            if (size<=m_capacity)
            {
                m_size=size;
            }
            else
            {
                soa_realloc(m_view,size);
            }
        }
    }

    void free()
    {
        m_size=0;
        m_capacity=0;
        soa_free(m_view);
    }
};